function h = personal_plot_multi(x, Y, x_label_latex, y_label_latex, legend_labels, legend_position, pdf_name)
    % Impostazioni standard LaTeX
    set(0, 'DefaultTextInterpreter', 'latex')
    set(0, 'DefaultLegendInterpreter', 'latex')
    set(0, 'DefaultAxesTickLabelInterpreter', 'latex')
    lw = 2;
    
    % Figure generation
    h = figure('Renderer', 'painters', 'Position', [10 10 900 350]);
    
    % --- AGGIORNAMENTO COLORI ---
    % 'r' = [1 0 0], 'g' = [0 0.5 0], 'b' = [0 0 1]
    colors = {[1, 0, 0], [0, 0.5, 0], [0, 0, 1]}; 
    styles = {'-', '-', '-'}; % Linee continue per tutti
    
    hold on;
    for i = 1:size(Y, 2)
        idx = mod(i-1, 3) + 1; % Cicla tra R, G, B
        plot(x, Y(:, i), styles{idx}, 'Linewidth', lw, 'Color', colors{idx});
    end
    
    % Legend
    legend(legend_labels);
    legend('Location', legend_position, 'Orientation', 'horizontal', 'AutoUpdate', 'off')
    
    % Labels 
    xlabel(x_label_latex)
    ylabel(y_label_latex)
    set(gca, 'FontSize', 22);
    
    % Grid
    grid on
    box on
    
    % Options
    set(gcf, 'color', 'w');
    set(h, 'MenuBar', 'none');
    set(h, 'ToolBar', 'none');
    
    % Limits
    xlim([x(1) x(end)])
    ylim([min(Y(:)), max(Y(:))]);
    
    % Fix inner position
    set(gca, 'InnerPosition', [0.1400 0.32 0.82 0.55])
    annotation('rectangle', [0 0 1 1], 'Color', 'w');
    
    exportgraphics(h, pdf_name);
end